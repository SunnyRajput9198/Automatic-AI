# Python User Data API

A comprehensive Python API solution for fetching, creating, updating, and managing user data with both server-side and client-side implementations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Using the Client](#using-the-client)
- [API Endpoints](#api-endpoints)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

This project provides:
1. **Flask API Server** (`flask_api.py`) - RESTful API endpoints for user data management
2. **User API Module** (`user_api.py`) - Core business logic with SQLite database
3. **API Client** (`api_client.py`) - Python client for consuming the API

## Features

- ✅ CRUD operations (Create, Read, Update, Delete) for users
- ✅ SQLite database backend with persistent storage
- ✅ RESTful API endpoints with proper HTTP methods
- ✅ Error handling and validation
- ✅ JSON request/response format
- ✅ Pagination support
- ✅ Python client library for easy API consumption
- ✅ Comprehensive logging

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- pip package manager
- Basic understanding of REST APIs and HTTP methods
- Familiarity with Python and command-line operations

## Installation

### Step 1: Clone or Download the Project

```bash
cd your-project-directory
```

### Step 2: Create a Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- Flask (web framework)
- requests (HTTP client library)

## Running the API

### Starting the Flask Server

```bash
python flask_api.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

The API server is now running and listening on `http://localhost:5000`.

### Server Configuration

The Flask server runs with:
- **Host**: 127.0.0.1 (localhost)
- **Port**: 5000
- **Debug Mode**: Enabled (auto-reloads on code changes)

## Using the Client

### Basic Client Usage

```bash
python api_client.py
```

This will execute a series of example API calls demonstrating all CRUD operations.

### Client Features

The `api_client.py` provides:
- Session management with automatic connection pooling
- Error handling and retry logic
- Support for all HTTP methods (GET, POST, PUT, DELETE)
- Automatic JSON serialization/deserialization
- Timeout handling

### Example Client Code

```python
from api_client import APIClient

# Initialize client
client = APIClient(base_url='http://localhost:5000')

# Fetch all users
users = client.get('/api/users')
print(users)

# Fetch specific user
user = client.get('/api/users/1')
print(user)

# Create new user
new_user = client.post('/api/users', {
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})
print(new_user)

# Update user
updated = client.put('/api/users/1', {'age': 31})
print(updated)

# Delete user
deleted = client.delete('/api/users/1')
print(deleted)

# Close session
client.close()
```

## API Endpoints

### User Management

#### Get All Users
```
GET /api/users
```
Returns a list of all users.

**Response:**
```json
[
  {
    "id": 1,
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "age": 28,
    "created_at": "2024-01-15 10:30:00"
  }
]
```

#### Get User by ID
```
GET /api/users/<id>
```
Returns a specific user by ID.

**Response:**
```json
{
  "id": 1,
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "created_at": "2024-01-15 10:30:00"
}
```

#### Create User
```
POST /api/users
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30
}
```

**Response (201 Created):**
```json
{
  "id": 4,
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30,
  "created_at": "2024-01-15 11:00:00"
}
```

#### Update User
```
PUT /api/users/<id>
Content-Type: application/json

{
  "age": 31,
  "name": "John Smith"
}
```

**Response:**
```json
{
  "id": 4,
  "name": "John Smith",
  "email": "john@example.com",
  "age": 31,
  "created_at": "2024-01-15 11:00:00"
}
```

#### Delete User
```
DELETE /api/users/<id>
```

**Response (200 OK):**
```json
{
  "message": "User deleted successfully"
}
```

## Examples

### Using cURL

```bash
# Get all users
curl http://localhost:5000/api/users

# Get specific user
curl http://localhost:5000/api/users/1

# Create user
curl -X POST http://localhost:5000/api/users \
  -H 'Content-Type: application/json' \
  -d '{"name": "Jane Doe", "email": "jane@example.com", "age": 25}'

# Update user
curl -X PUT http://localhost:5000/api/users/1 \
  -H 'Content-Type: application/json' \
  -d '{"age": 29}'

# Delete user
curl -X DELETE http://localhost:5000/api/users/1
```

### Using Python Requests

```python
import requests

BASE_URL = 'http://localhost:5000'

# Get all users
response = requests.get(f'{BASE_URL}/api/users')
print(response.json())

# Create user
data = {'name': 'Jane Doe', 'email': 'jane@example.com', 'age': 25}
response = requests.post(f'{BASE_URL}/api/users', json=data)
print(response.json())

# Update user
update_data = {'age': 26}
response = requests.put(f'{BASE_URL}/api/users/1', json=update_data)
print(response.json())

# Delete user
response = requests.delete(f'{BASE_URL}/api/users/1')
print(response.json())
```

## Best Practices

### Security
- ✅ Always validate input data on the server side
- ✅ Use HTTPS in production environments
- ✅ Implement authentication and authorization
- ✅ Sanitize user inputs to prevent SQL injection
- ✅ Use environment variables for sensitive configuration

### Performance
- ✅ Implement caching for frequently accessed data
- ✅ Use database indexing on frequently queried fields
- ✅ Implement pagination for large datasets
- ✅ Use connection pooling for database connections
- ✅ Monitor API response times

### Error Handling
- ✅ Return appropriate HTTP status codes
- ✅ Provide meaningful error messages
- ✅ Log all errors for debugging
- ✅ Implement retry logic for transient failures
- ✅ Handle timeouts gracefully

### Testing
- ✅ Write unit tests for business logic
- ✅ Write integration tests for API endpoints
- ✅ Test error scenarios and edge cases
- ✅ Use tools like pytest for testing
- ✅ Maintain test coverage above 80%

### Documentation
- ✅ Keep API documentation up-to-date
- ✅ Document all endpoints and parameters
- ✅ Provide example requests and responses
- ✅ Include error codes and their meanings
- ✅ Use tools like Swagger/OpenAPI for API documentation

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

### Database Locked
If you get a "database is locked" error:
```bash
# Delete the database file and restart
rm users.db
python flask_api.py
```

### Connection Refused
Ensure the Flask server is running:
```bash
python flask_api.py
```

### Module Not Found
Install missing dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── flask_api.py          # Flask server with API endpoints
├── user_api.py           # Core user data management logic
├── api_client.py         # Python client for API consumption
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── users.db              # SQLite database (created on first run)
```

## Next Steps

1. **Add Authentication**: Implement JWT or OAuth2 authentication
2. **Add Database Migrations**: Use Alembic for schema versioning
3. **Add API Documentation**: Use Swagger/OpenAPI
4. **Add Unit Tests**: Create comprehensive test suite
5. **Deploy to Production**: Use Docker and cloud platforms
6. **Add Monitoring**: Implement logging and monitoring
7. **Add Rate Limiting**: Prevent API abuse

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API endpoint documentation
3. Check server logs for error messages
4. Verify all dependencies are installed

## License

This project is provided as-is for educational purposes.

---

**Last Updated**: 2024
**Version**: 1.0.0
