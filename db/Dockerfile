# Use the official PostgreSQL image from the Docker Hub
FROM postgres:latest

# Set environment variables for PostgreSQL
# You can change these to your preferred username, password, and database name
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydatabase

# Expose the port that PostgreSQL runs on
EXPOSE 5432

# Health check for the database service
# This periodically tests the connection to the database to ensure it's running
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD pg_isready -U $POSTGRES_USER
