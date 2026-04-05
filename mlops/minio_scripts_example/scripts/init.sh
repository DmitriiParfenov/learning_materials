#! /bin/bash
# Подключаем инстанс
echo "Подключаем наш инстанс Minio"
mc alias set minio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Создаём бакеты
echo "Создаём бакеты"
mc mb minio/machine_learning
sleep 3

# Создаём пользователей
echo "Создаём пользователей"
mc admin user add minio ${APP_USER_RW} ${APP_PASS_RW}
sleep 3
mc admin user add minio ${APP_USER_RO} ${APP_PASS_RO}
sleep 3

# Применяем полтитики
echo "Создаём политики"
mc admin policy create minio plastilin-rw-policy /policy/rw_policy.json
sleep 3
mc admin policy create minio plastilin-ro-policy /policy/ro_policy.json
sleep 3

# Назначаем политики
echo "Назначаем политики пользователям."
mc admin policy attach minio plastilin-rw-policy --user plastilin-app-rw
sleep 3
mc admin policy attach minio plastilin-ro-policy --user plastilin-app-ro
