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

# numpy and pandas must be installed before scikit-learn: sklearn's pyproject.toml
# build backend tries to import numpy during metadata preparation, which fails on
# Alpine (musl/no binary wheels) if numpy isn't already present.
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pandas

RUN pip3 install --no-cache-dir --break-system-packages \
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