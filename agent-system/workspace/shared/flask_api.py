from flask import Flask, jsonify

app = Flask(__name__)

# Sample user data
users = [
    {
        'id': 1,
        'name': 'Alice Johnson',
        'email': 'alice@example.com',
        'age': 28
    },
    {
        'id': 2,
        'name': 'Bob Smith',
        'email': 'bob@example.com',
        'age': 35
    },
    {
        'id': 3,
        'name': 'Carol White',
        'email': 'carol@example.com',
        'age': 31
    },
    {
        'id': 4,
        'name': 'David Brown',
        'email': 'david@example.com',
        'age': 42
    }
]

@app.route('/users', methods=['GET'])
def get_users():
    """Return all users as JSON"""
    return jsonify({
        'status': 'success',
        'data': users,
        'count': len(users)
    }), 200

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Return a specific user by ID"""
    user = next((u for u in users if u['id'] == user_id), None)
    if user:
        return jsonify({
            'status': 'success',
            'data': user
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'User with ID {user_id} not found'
        }), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask API is running'
    }), 200

if __name__ == '__main__':
    print('Flask API with /users endpoint created successfully!')
    print('Sample endpoints:')
    print('  GET /users - Returns all users')
    print('  GET /users/<id> - Returns specific user')
    print('  GET /health - Health check')
    print('\nTo run the API, execute: python flask_api.py')
    print('Then visit: http://localhost:5000/users')
