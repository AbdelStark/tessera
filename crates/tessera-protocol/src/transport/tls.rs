// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mutually authenticated TLS 1.3 over host-owned, already-connected TCP sockets.
//! DNS, listening, connecting and identity storage remain the host's responsibility.

use std::{
    io::{self, Read, Write},
    net::{Shutdown, TcpStream},
    sync::Arc,
    time::{Duration, Instant},
};

use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection};
use tessera_codec::{
    DecodeLimits, envelope,
    error::{ErrorCode, TesseraError},
};

use super::{
    Deadlines, TimedWriter, check_len, deadline, encoded_len, error, io_error, live, read_bytes,
    remaining,
};
use crate::Message;

/// Client identity and trust configuration. Requires a client certificate and
/// verifies the server's certificate and DNS/IP name against explicit trust roots.
#[derive(Clone)]
pub struct ClientTlsConfig(Arc<ClientConfig>);

impl ClientTlsConfig {
    /// Builds TLS-1.3-only configuration, using rustls's ring provider. Identity
    /// keys move into the TLS stack and are never formatted by this API.
    pub fn new(
        roots: RootCertStore,
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
    ) -> Result<Self, TesseraError> {
        let mut config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_protocol_versions(&[&rustls::version::TLS13])
                .map_err(|_| error(ErrorCode::Usage))?
                .with_root_certificates(roots)
                .with_client_auth_cert(chain, key)
                .map_err(|_| error(ErrorCode::Usage))?;
        // Always authenticate this connection, rather than relying on a previous session.
        config.resumption = rustls::client::Resumption::disabled();
        Ok(Self(Arc::new(config)))
    }
}

/// Server configuration with mandatory client authentication. There is no
/// anonymous-client or TLS-1.2 fallback constructor.
#[derive(Clone)]
pub struct ServerTlsConfig(Arc<ServerConfig>);

impl ServerTlsConfig {
    /// Builds TLS-1.3-only configuration with explicit client trust roots.
    pub fn new(
        roots: RootCertStore,
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
    ) -> Result<Self, TesseraError> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(roots),
            provider.clone(),
        )
        .build()
        .map_err(|_| error(ErrorCode::Usage))?;
        let mut config = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|_| error(ErrorCode::Usage))?
            .with_client_cert_verifier(verifier)
            .with_single_cert(chain, key)
            .map_err(|_| error(ErrorCode::Usage))?;
        config.send_tls13_tickets = 0;
        Ok(Self(Arc::new(config)))
    }
}

struct TimedSocket {
    socket: TcpStream,
    end: Instant,
}
impl Read for TimedSocket {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.socket.set_read_timeout(Some(remaining(self.end)?))?;
        let n = self.socket.read(bytes)?;
        remaining(self.end)?;
        Ok(n)
    }
}
impl Write for TimedSocket {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.socket.set_write_timeout(Some(remaining(self.end)?))?;
        let n = self.socket.write(bytes)?;
        remaining(self.end)?;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        remaining(self.end)?;
        Ok(())
    }
}

enum Peer {
    Client(ClientConnection),
    Server(ServerConnection),
}
trait Channel: Read + Write {}
impl<T: Read + Write> Channel for T {}

/// One authenticated Live session. Frames are `u64_le(length) || TCE envelope`.
/// Prefixes, TLS records and certificate details never enter the transcript.
///
/// A failed operation shuts down the socket and makes all future calls return
/// `ERR-PROTO-ORDER`. Drop the associated prover session on failure (INV-PROTO-1).
/// TLS/configuration errors are sanitized to `ERR-RES-IO`/`ERR-USAGE`; their
/// underlying strings, certificates, keys and message bytes are not retained in errors.
pub struct TlsTransport {
    peer: Peer,
    io: TimedSocket,
    limits: DecodeLimits,
    deadlines: Deadlines,
    failed: bool,
}

impl TlsTransport {
    /// Authenticates the server on an already-connected socket and presents the
    /// configured client certificate. The host must bound TCP connect/DNS itself.
    pub fn client(
        socket: TcpStream,
        name: ServerName<'static>,
        config: ClientTlsConfig,
        limits: DecodeLimits,
        deadlines: Deadlines,
        handshake_timeout: Duration,
    ) -> Result<Self, TesseraError> {
        let mut connection =
            ClientConnection::new(config.0, name).map_err(|_| error(ErrorCode::Usage))?;
        connection.set_buffer_limit(Some(65536));
        Self::start(
            socket,
            Peer::Client(connection),
            limits,
            deadlines,
            handshake_timeout,
        )
    }

    /// Authenticates a client on a socket the host accepted. A client with no
    /// certificate, an untrusted certificate or TLS 1.2 alone is refused.
    pub fn server(
        socket: TcpStream,
        config: ServerTlsConfig,
        limits: DecodeLimits,
        deadlines: Deadlines,
        handshake_timeout: Duration,
    ) -> Result<Self, TesseraError> {
        let mut connection =
            ServerConnection::new(config.0).map_err(|_| error(ErrorCode::Usage))?;
        connection.set_buffer_limit(Some(65536));
        Self::start(
            socket,
            Peer::Server(connection),
            limits,
            deadlines,
            handshake_timeout,
        )
    }

    fn start(
        socket: TcpStream,
        peer: Peer,
        limits: DecodeLimits,
        deadlines: Deadlines,
        timeout: Duration,
    ) -> Result<Self, TesseraError> {
        socket.set_nonblocking(false).map_err(io_error)?;
        let mut transport = Self {
            peer,
            io: TimedSocket {
                socket,
                end: deadline(timeout)?,
            },
            limits,
            deadlines,
            failed: false,
        };
        let result = match &mut transport.peer {
            Peer::Client(c) => c.complete_io(&mut transport.io),
            Peer::Server(c) => c.complete_io(&mut transport.io),
        };
        result.map_err(io_error)?;
        remaining(transport.io.end).map_err(io_error)?;
        Ok(transport)
    }

    fn with_stream<R>(
        &mut self,
        f: impl FnOnce(&mut dyn Channel) -> Result<R, TesseraError>,
    ) -> Result<R, TesseraError> {
        if self.failed {
            return Err(error(ErrorCode::ProtoOrder));
        }
        let result = match &mut self.peer {
            Peer::Client(c) => f(&mut rustls::Stream::new(c, &mut self.io)),
            Peer::Server(c) => f(&mut rustls::Stream::new(c, &mut self.io)),
        };
        if result.is_err() {
            self.failed = true;
            let _ = self.io.socket.shutdown(Shutdown::Both);
        }
        result
    }

    fn begin(&mut self, duration: Duration) -> Result<Instant, TesseraError> {
        if self.failed {
            return Err(error(ErrorCode::ProtoOrder));
        }
        match deadline(duration) {
            Ok(end) => {
                self.io.end = end;
                Ok(end)
            }
            Err(e) => {
                self.failed = true;
                let _ = self.io.socket.shutdown(Shutdown::Both);
                Err(e)
            }
        }
    }

    /// Sends a bounded length prefix and the unchanged canonical envelope.
    pub fn send<T: Message>(&mut self, message: &T) -> Result<(), TesseraError> {
        let end = self.begin(self.deadlines.for_message(T::KIND))?;
        let limits = self.limits;
        self.with_stream(|stream| {
            remaining(end).map_err(io_error)?;
            let len = encoded_len(message, limits)?;
            stream.write_all(&len.to_le_bytes()).map_err(io_error)?;
            let mut out = TimedWriter { inner: stream, end };
            envelope::write(&mut out, &T::KIND.schema(), message).map_err(io_error)?;
            out.flush().map_err(io_error)?;
            remaining(end).map_err(io_error)?;
            Ok(())
        })
    }

    /// Reads the prefix before allocating, then decodes exactly that envelope.
    /// A slow peer cannot extend the absolute message deadline by sending bytes.
    pub fn receive<T: Message>(&mut self) -> Result<T, TesseraError> {
        let end = self.begin(self.deadlines.for_message(T::KIND))?;
        let limits = self.limits;
        self.with_stream(|mut stream| {
            live(T::KIND)?;
            remaining(end).map_err(io_error)?;
            let mut prefix = [0; 8];
            stream.read_exact(&mut prefix).map_err(io_error)?;
            let len = check_len(u64::from_le_bytes(prefix), limits)?;
            let bytes = read_bytes(&mut stream, len, end)?;
            let value = T::open(&bytes, limits).map_err(TesseraError::from)?;
            remaining(end).map_err(io_error)?;
            Ok(value)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{
        DirectoryTransport, filename,
        tests::{Directory, message},
    };
    use crate::{SeedCommit, SeedReveal};
    use rcgen::{
        BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    };
    use std::{net::TcpListener, thread};

    fn configs() -> (ClientTlsConfig, ServerTlsConfig, RootCertStore) {
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let key = KeyPair::generate().unwrap();
        let ca = params.self_signed(&key).unwrap();
        let issuer = Issuer::new(params, key);
        let mut roots = RootCertStore::empty();
        roots.add(ca.der().clone()).unwrap();
        let leaf = |name: &str| {
            let mut params = CertificateParams::new(vec![name.to_owned()]).unwrap();
            params.extended_key_usages = vec![
                ExtendedKeyUsagePurpose::ServerAuth,
                ExtendedKeyUsagePurpose::ClientAuth,
            ];
            let key = KeyPair::generate().unwrap();
            let cert = params.signed_by(&key, &issuer).unwrap();
            (
                vec![cert.der().clone()],
                PrivateKeyDer::try_from(key.serialize_der()).unwrap(),
            )
        };
        let (cert, key) = leaf("localhost");
        let server = ServerTlsConfig::new(roots.clone(), cert, key).unwrap();
        let (cert, key) = leaf("client.local");
        let client = ClientTlsConfig::new(roots.clone(), cert, key).unwrap();
        (client, server, roots)
    }

    fn budgets() -> Deadlines {
        Deadlines {
            round2: Duration::from_secs(2),
            other: Duration::from_secs(2),
        }
    }
    fn server(socket: TcpStream, config: ServerTlsConfig) -> Result<TlsTransport, TesseraError> {
        TlsTransport::server(
            socket,
            config,
            DecodeLimits {
                max_total_bytes: 4096,
                ..DecodeLimits::MESSAGE
            },
            budgets(),
            Duration::from_secs(2),
        )
    }
    fn client(socket: TcpStream, config: ClientTlsConfig) -> Result<TlsTransport, TesseraError> {
        TlsTransport::client(
            socket,
            ServerName::try_from("localhost").unwrap(),
            config,
            DecodeLimits {
                max_total_bytes: 4096,
                ..DecodeLimits::MESSAGE
            },
            budgets(),
            Duration::from_secs(2),
        )
    }

    #[test]
    fn stream_frames_are_exactly_directory_envelopes_and_messages_do_not_coalesce() {
        let dir = Directory::new();
        let mut directory =
            DirectoryTransport::new(&dir.0, DecodeLimits::MESSAGE, budgets()).unwrap();
        directory.send(&message()).unwrap();
        let expected = std::fs::read(
            dir.0
                .join(filename(crate::MessageKind::SeedCommit).unwrap()),
        )
        .unwrap();
        let (client_config, server_config, _) = configs();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let task = thread::spawn(move || {
            let mut peer = server(listener.accept().unwrap().0, server_config).unwrap();
            match &peer.peer {
                Peer::Server(c) => {
                    assert_eq!(c.protocol_version(), Some(rustls::ProtocolVersion::TLSv1_3))
                }
                _ => unreachable!(),
            }
            peer.with_stream(|stream| {
                let mut prefix = [0; 8];
                stream.read_exact(&mut prefix).map_err(io_error)?;
                assert_eq!(u64::from_le_bytes(prefix), expected.len() as u64);
                let mut bytes = vec![0; expected.len()];
                stream.read_exact(&mut bytes).map_err(io_error)?;
                assert_eq!(bytes, expected);
                Ok(())
            })
            .unwrap();
            assert_eq!(peer.receive::<SeedCommit>().unwrap(), message());
            peer.send(&SeedReveal {
                session_id: [1; 16],
                share: [7; 32],
            })
            .unwrap();
        });
        let mut peer = client(TcpStream::connect(address).unwrap(), client_config).unwrap();
        peer.send(&message()).unwrap();
        peer.send(&message()).unwrap();
        assert_eq!(peer.receive::<SeedReveal>().unwrap().share, [7; 32]);
        task.join().unwrap();
    }

    #[test]
    fn clients_without_a_certificate_or_with_an_untrusted_certificate_are_refused() {
        for authenticated in [false, true] {
            let (_, server_config, roots) = configs();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let task =
                thread::spawn(
                    move || match server(listener.accept().unwrap().0, server_config) {
                        Err(e) => assert_eq!(e.code, ErrorCode::ResIo),
                        Ok(_) => panic!("untrusted or anonymous client completed authentication"),
                    },
                );
            let config = ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_root_certificates(roots);
            let config = if authenticated {
                // This identity belongs to a separate, untrusted CA.
                let (other, _, _) = configs();
                config.with_client_cert_resolver(other.0.client_auth_cert_resolver.clone())
            } else {
                config.with_no_client_auth()
            };
            let mut socket = TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut connection =
                ClientConnection::new(Arc::new(config), ServerName::try_from("localhost").unwrap())
                    .unwrap();
            let mut stream = rustls::Stream::new(&mut connection, &mut socket);
            let _ = stream.write_all(b"not a protocol frame");
            assert!(stream.read(&mut [0; 1]).is_err());
            task.join().unwrap();
        }
    }

    #[test]
    fn wrong_server_name_and_handshake_timeout_are_refused() {
        let (client_config, server_config, _) = configs();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let task = thread::spawn(move || {
            assert!(server(listener.accept().unwrap().0, server_config).is_err());
        });
        let result = TlsTransport::client(
            TcpStream::connect(address).unwrap(),
            ServerName::try_from("wrong.local").unwrap(),
            client_config.clone(),
            DecodeLimits::MESSAGE,
            budgets(),
            Duration::from_secs(2),
        );
        assert_eq!(result.err().unwrap().code, ErrorCode::ResIo);
        task.join().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let socket = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let _silent = listener.accept().unwrap().0;
        let start = Instant::now();
        assert_eq!(
            TlsTransport::client(
                socket,
                ServerName::try_from("localhost").unwrap(),
                client_config,
                DecodeLimits::MESSAGE,
                budgets(),
                Duration::from_millis(30)
            )
            .err()
            .unwrap()
            .code,
            ErrorCode::ProtoTimeout
        );
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn malformed_prefixes_and_envelopes_refuse_exactly_and_end_the_transport() {
        let mut corrupt = message().seal();
        *corrupt.last_mut().unwrap() ^= 1;
        let mut wrong_schema = SeedReveal {
            session_id: [1; 16],
            share: [3; 32],
        }
        .seal();
        // Both frames are well-formed envelopes; only one names the expected schema.
        let cases = [
            (u64::MAX.to_le_bytes().to_vec(), ErrorCode::CodecLimit),
            (0u64.to_le_bytes().to_vec(), ErrorCode::CodecTruncated),
            (
                [(corrupt.len() as u64).to_le_bytes().to_vec(), corrupt].concat(),
                ErrorCode::CodecDigest,
            ),
            (
                [
                    (wrong_schema.len() as u64).to_le_bytes().to_vec(),
                    std::mem::take(&mut wrong_schema),
                ]
                .concat(),
                ErrorCode::CodecMagic,
            ),
            (
                [100u64.to_le_bytes().as_slice(), &[1, 2]].concat(),
                ErrorCode::ResIo,
            ),
        ];
        for (frame, code) in cases {
            let (client_config, server_config, _) = configs();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let task = thread::spawn(move || {
                let mut peer = server(listener.accept().unwrap().0, server_config).unwrap();
                assert_eq!(peer.receive::<SeedCommit>().unwrap_err().code, code);
                assert_eq!(
                    peer.receive::<SeedCommit>().unwrap_err().code,
                    ErrorCode::ProtoOrder
                );
                assert_eq!(
                    peer.send(&message()).unwrap_err().code,
                    ErrorCode::ProtoOrder
                );
            });
            let mut peer = client(TcpStream::connect(address).unwrap(), client_config).unwrap();
            peer.with_stream(|stream| {
                stream.write_all(&frame).map_err(io_error)?;
                stream.flush().map_err(io_error)
            })
            .unwrap();
            drop(peer);
            task.join().unwrap();
        }
    }

    #[test]
    fn failed_send_never_allows_the_transport_to_be_reused() {
        for limit in [true, false] {
            let (client_config, server_config, _) = configs();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let task = thread::spawn(move || {
                let mut peer = server(listener.accept().unwrap().0, server_config).unwrap();
                if limit {
                    peer.limits.max_total_bytes = 99;
                } else {
                    peer.deadlines.other = Duration::ZERO;
                }
                assert_eq!(
                    peer.send(&message()).unwrap_err().code,
                    if limit {
                        ErrorCode::CodecLimit
                    } else {
                        ErrorCode::ProtoTimeout
                    }
                );
                assert_eq!(
                    peer.send(&message()).unwrap_err().code,
                    ErrorCode::ProtoOrder
                );
                assert_eq!(
                    peer.receive::<SeedCommit>().unwrap_err().code,
                    ErrorCode::ProtoOrder
                );
            });
            let mut peer = client(TcpStream::connect(address).unwrap(), client_config).unwrap();
            assert!(peer.receive::<SeedCommit>().is_err());
            task.join().unwrap();
        }
    }

    #[test]
    fn a_dripping_peer_cannot_reset_the_message_deadline() {
        let (client_config, server_config, _) = configs();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let task = thread::spawn(move || {
            let mut peer = server(listener.accept().unwrap().0, server_config).unwrap();
            peer.deadlines.other = Duration::from_millis(70);
            let start = Instant::now();
            assert_eq!(
                peer.receive::<SeedCommit>().unwrap_err().code,
                ErrorCode::ProtoTimeout
            );
            assert!(start.elapsed() < Duration::from_secs(1));
        });
        let mut peer = client(TcpStream::connect(address).unwrap(), client_config).unwrap();
        let frame = [100u64.to_le_bytes().to_vec(), message().seal()].concat();
        for byte in frame.into_iter().take(20) {
            if peer
                .with_stream(|stream| {
                    stream.write_all(&[byte]).map_err(io_error)?;
                    stream.flush().map_err(io_error)
                })
                .is_err()
            {
                break;
            }
            thread::sleep(Duration::from_millis(15));
        }
        task.join().unwrap();
    }
}
