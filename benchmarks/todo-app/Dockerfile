# Todo-app overlay on the scenario-agnostic toolchain (infra/toolchain/Dockerfile): adds the
# .NET SDK that the todo-app phases (build-fix loop) and gates (build + score) need. Other
# scenarios that don't need .NET (e.g. smoke) just use the base image directly.
#
# Build order (the orchestrator picks this image automatically for the todo-app scenario):
#   docker build -t dail-toolchain          -f infra/toolchain/Dockerfile .
#   docker build -t dail-toolchain-todo-app -f benchmarks/todo-app/Dockerfile .
FROM dail-toolchain

# .NET SDK 10 from Microsoft's package feed (pulls the ICU/SSL/etc. runtime deps that the
# bare debian base doesn't carry — the previous all-in-one image got these from the SDK base).
RUN curl -fsSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -o /tmp/ms-prod.deb \
    && dpkg -i /tmp/ms-prod.deb && rm /tmp/ms-prod.deb \
    && apt-get update && apt-get install -y --no-install-recommends dotnet-sdk-10.0 \
    && rm -rf /var/lib/apt/lists/*
