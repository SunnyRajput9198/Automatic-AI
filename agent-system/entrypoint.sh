#!/bin/sh
set -e

echo "🚀 Starting Agent System"

echo "📦 Running database migrations..."
alembic upgrade head

echo "✅ Migrations complete"

echo "🌐 Starting API..."
exec "$@"
#!/bin/sh
set -e

echo "🚀 Starting Agent System"

echo "📦 Running database migrations..."
alembic upgrade head

echo "✅ Migrations complete"

echo "🌐 Starting API..."
exec "$@"
