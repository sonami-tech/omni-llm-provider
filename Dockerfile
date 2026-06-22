FROM rust:bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build --release -p omni

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/omni /usr/local/bin/omni

ENV OMNI_BIND=0.0.0.0
EXPOSE 18321

ENTRYPOINT ["omni"]