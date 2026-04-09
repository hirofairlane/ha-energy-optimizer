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
    g++

RUN pip3 install --no-cache-dir --break-system-packages \
    scikit-learn \
    requests \
    flask \
    apscheduler \
    pandas \
    joblib

COPY rootfs /

RUN chmod a+x /usr/bin/run.sh \
    && chmod a+x /usr/bin/energy_optimizer.py

CMD ["/usr/bin/run.sh"]