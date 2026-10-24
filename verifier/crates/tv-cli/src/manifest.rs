// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's independent run-manifest/1 writer (05 §6, RFC-0001).
use crate::{
    files,
    observability::{Observation, optional, quoted, timestamp},
};
use std::path::Path;
use std::time::Duration;
use tv_core::verdict::ErrorCode;

pub(crate) fn write(
    path: &Path,
    args: &[String],
    observation: &Observation,
    elapsed: Duration,
    exit_code: i32,
    error: Option<&str>,
    out: Option<&str>,
) -> Result<(), ErrorCode> {
    let inputs = observation
        .inputs
        .iter()
        .map(|a| {
            format!(
                "{{\"path\":{},\"schema\":{},\"class\":{},\"digest\":{}}}",
                quoted(&a.path),
                quoted(a.schema),
                quoted(a.class),
                quoted(&a.digest)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    // JSON verdict reports have no envelope digest in the committed schema.
    let outputs = out.map_or(String::new(), |p| {
        format!(
            "{{\"path\":{},\"schema\":\"tessera.verdict/1\",\"class\":\"public\",\"digest\":null}}",
            quoted(p)
        )
    });
    let mut arguments = vec![quoted("verify")];
    arguments.extend(args.iter().map(|s| quoted(s)));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let json = format!(
        concat!(
            "{{\"schema\":\"tessera.run-manifest/1\",",
            "\"tool\":{{\"name\":\"tessera-verify\",\"version\":{},\"git_commit\":{},\"profile\":\"{}\",\"features\":[{}]}},",
            "\"toolchain\":{{\"rustc\":{},\"cuda_runtime\":null,\"cuda_driver\":null,\"python\":null,\"torch\":null}},",
            "\"host\":{{\"os\":{},\"arch\":{},\"cpu_model\":null,\"cores\":{},\"memory_bytes\":null,\"gpus\":[],\"ecc\":null}},",
            "\"command\":\"verify\",\"arguments\":[{}],\"params\":{},\"protocol\":\"tessera-protocol/1\",",
            "\"inputs\":[{}],\"outputs\":[{}],\"public_seeds\":[],\"started_at\":{},\"elapsed_seconds\":{},",
            "\"metrics\":[{}],\"exit_code\":{},\"error_code\":{}}}\n"
        ),
        quoted(env!("CARGO_PKG_VERSION")),
        optional(option_env!("TESSERA_VERIFIER_BUILD_COMMIT")),
        profile,
        if tv_core::params::ParamSet::ALL
            .iter()
            .any(|p| p.name() == "test-small/1")
        {
            quoted("tv-core/test-params")
        } else {
            String::new()
        },
        optional(option_env!("TESSERA_VERIFIER_BUILD_RUSTC")),
        quoted(std::env::consts::OS),
        quoted(std::env::consts::ARCH),
        std::thread::available_parallelism().map_or("null".into(), |n| n.get().to_string()),
        arguments.join(","),
        optional(observation.params.as_deref()),
        inputs,
        outputs,
        quoted(&timestamp(observation.started)),
        elapsed.as_secs_f64(),
        observation.metrics(),
        exit_code,
        optional(error)
    );
    files::write(path, json.as_bytes())
}
