#! /bin/bash
# Подключаем инстанс
echo "Подключаем наш инстанс Minio"
mc alias set minio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Создаём бакеты
echo "Создаём бакеты"
mc mb minio/${MLFLOW_BUCKET_NAME}
sleep 3
exit 0
