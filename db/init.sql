-- Create the dblink extension if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'dblink') THEN
      CREATE EXTENSION dblink;
   END IF;
END
$$;

-- Create user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'myuser') THEN
      CREATE USER myuser WITH PASSWORD 'mypassword';
   END IF;
END
$$;

-- Create database if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mydatabase') THEN
      CREATE DATABASE mydatabase;
   END IF;
END
$$;

-- Grant privileges to the user on the database
GRANT ALL PRIVILEGES ON DATABASE mydatabase TO myuser;

-- Connect to the newly created or existing database and create the table.
\c mydatabase

-- Create the 'conversations' table if it doesn't exist
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conversations') THEN
      CREATE TABLE conversations (
          id SERIAL PRIMARY KEY,
          timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          prompt TEXT NOT NULL,
          response TEXT NOT NULL
      );
   END IF;
END
$$;