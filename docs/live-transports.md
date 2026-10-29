# Live transports on the prover side

[RFC-0014](rfcs/RFC-0014-interaction-and-soundness.md#live-transport) defines
directory exchange and mutually authenticated TLS 1.3. The protocol library's
`transport` module implements both, using the existing message codecs. Transport
framing never enters the Fiat–Shamir transcript.

## Directory exchange

`tessera_protocol::transport::DirectoryTransport` uses an existing,
operator-controlled directory and caller-supplied `DecodeLimits` and `Deadlines`.
Call `send(&message)` or `receive::<MessageType>()`. Each file holds exactly the
TCE envelope, without a length prefix. Filenames match the CLI's existing exchange:

| Step | File |
|---|---|
| 0 | `00-SeedCommit.tce` |
| 1 | `01-Round1.tce`, `01-ProverSeedShare.tce` |
| 2 | `02-SeedReveal.tce` |
| 3 | `03-Round2Response.tce` |
| 4 | `04-ColumnQueries.tce` |
| 5 | `05-Round3Response.tce` |

Publication writes a fresh exclusive temporary file, flushes it, links the final
name without replacement and syncs the directory. Unix files are created with
mode 0600. Receives refuse non-files and symlinks, bound the length before
allocation, and validate the envelope and its typed payload. Unix opens use
`O_NOFOLLOW | O_NONBLOCK` to refuse a symlink or FIFO swapped into the final name.
The directory's parent and access permissions remain the operator's responsibility.
Transfer only complete files into a receiving directory.

## TLS exchange

Enable the prover library's `tls` feature:

```toml
tessera-protocol = { path = "crates/tessera-protocol", features = ["tls"] }
```

Use `transport::tls::{ClientTlsConfig, ServerTlsConfig, TlsTransport}`. Build an
identity configuration with explicit trust roots, a certificate chain and its
private key using rustls's DER types. Keep private-key loading and storage in
the host's identity manager. The configuration consumes the key; this API never
formats it. Both configurations use rustls 0.23's ring provider, admit TLS 1.3
only, and disable session resumption. The server requires an authenticated client
certificate. The client presents its certificate and verifies the server's
certificate chain and DNS/IP name. No insecure-verifier constructor is exposed.

The host owns DNS, TCP connection/listener setup and the deadlines for those
operations. Give an already-connected socket to `TlsTransport::client` or
`TlsTransport::server`, along with configuration, decoder limits, message
deadlines and a separate handshake timeout. These constructors complete the
authenticated handshake before returning. A frame is a little-endian `u64`
length followed by exactly that many bytes of the original TCE envelope.
`send` and `receive` have the same typed interface as the directory transport.
The length is checked before allocation. Buffers grow only as body bytes arrive,
so a length prefix alone cannot force a whole-frame allocation. TLS has its own
bounded record buffers.

The default library build does not link rustls. The `tessera prove --live DIR`
CLI retains its directory interface; network service setup and the independently
implemented verifier transport are outside this library API
([#181](https://github.com/AbdelStark/tessera/issues/181)).

## Deadlines, errors and session lifetime

`Deadlines::default()` gives `Round2Response` one hour and other messages ten
minutes (OQ-0014-1). Each operation has one absolute deadline, including framing
and all I/O. Receiving a byte does not restart it. TLS applies the remaining
budget to every underlying socket read and write, including handshake work.
Filesystem deadlines are cooperative: a blocked kernel/filesystem operation
cannot be preempted by this synchronous API.

Timeouts report `ERR-PROTO-TIMEOUT`, I/O and TLS authentication failures
`ERR-RES-IO`, invalid configuration `ERR-USAGE`, and decoding failures their
existing `ERR-CODEC-*` codes. Errors retain no I/O paths, certificate diagnostics,
private keys or message bytes. Limits are host policy and should fit the host's
actual memory budget. NI proofs are refused by these Live APIs.

Use one transport for one session. Every failed operation poisons that transport;
later calls return `ERR-PROTO-ORDER`. TLS also shuts down its socket. Drop the
associated prover session immediately on any transport failure so INV-PROTO-1
discards its commitment and randomness. Never retry that session with another
peer, transport or seed. The protocol state machines still enforce message order;
the transport validates delivery and encoding, rather than interpreting proof checks.

Run the transport tests with:

```sh
cargo test -p tessera-protocol --features tls --lib transport --locked
```
