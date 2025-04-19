# Use the official PostgreSQL image from the Docker Hub
FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydatabase

# Copy the initialization script
COPY ./init.sql /docker-entrypoint-initdb.d/

# Expose the port that PostgreSQL runs on
EXPOSE 5432