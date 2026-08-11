ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    python3-dev \
    musl-dev \
    g++ \
    openblas-dev \
    gfortran

# numpy/pandas/scipy and the meson-python build backend must be installed before
# scikit-learn, which must then build with --no-build-isolation: sklearn's
# pyproject.toml build backend imports numpy during metadata preparation, but pip's
# isolated build env hides system site-packages, so a pre-installed numpy is invisible
# unless isolation is disabled — and disabling isolation means the build backend
# itself (meson-python, Cython, ninja) must also already be present system-wide.
#
# numpy must stay below 2.4: musllinux wheels from 2.4 onward are built with an
# x86-64-v2 baseline (SSE4.2/POPCNT), which crashes with "NumPy was built with
# baseline optimizations (X86_V2) but your machine doesn't support" on VMs using a
# generic/older QEMU CPU type (e.g. Proxmox default "kvm64") — a common HA VM setup.
RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy<2.4" \
    scipy \
    pandas \
    meson-python \
    Cython \
    ninja \
    pybind11

RUN pip3 install --no-cache-dir --break-system-packages --no-build-isolation \
    scikit-learn \
    requests \
    flask \
    apscheduler \
    joblib \
    pymysql

COPY rootfs /
COPY config.yaml /addon-config.yaml

RUN chmod a+x /usr/bin/run.sh \
    && chmod a+x /usr/bin/energy_optimizer.py

CMD ["/usr/bin/run.sh"]