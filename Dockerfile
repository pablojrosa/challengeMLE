# Usar la versión específica de Python como base
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

# Establecer el directorio de trabajo en el contenedor
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Esponer el puerto a utilizar
EXPOSE 8080

# Comando para iniciar la aplicación usando Uvicorn con configuración para producción
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]