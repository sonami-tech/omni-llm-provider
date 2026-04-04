FROM rust:bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl ca-certificates \
	&& curl -fsSL https://claude.ai/install.sh | bash \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

COPY --from=builder /build/target/release/claude-code-provider /usr/local/bin/

ENV CCP_HOST=0.0.0.0
ENV CCP_NO_ISOLATE=true
EXPOSE 18321

ENTRYPOINT ["claude-code-provider"]
