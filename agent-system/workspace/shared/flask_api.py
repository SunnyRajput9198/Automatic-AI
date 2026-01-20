from flask import Flask, jsonify

app = Flask(__name__)

# Sample user data
users = [
    {'id': 1, 'name': 'Alice Johnson', 'email': 'alice@example.com', 'role': 'admin'},
    {'id': 2, 'name': 'Bob Smith', 'email': 'bob@example.com', 'role': 'user'},
    {'id': 3, 'name': 'Carol White', 'email': 'carol@example.com', 'role': 'user'},
    {'id': 4, 'name': 'David Brown', 'email': 'david@example.com', 'role': 'moderator'}
]

@app.route('/api/users', methods=['GET'])
def get_users():
    """GET endpoint that returns all sample user data"""
    return jsonify({
        'status': 'success',
        'data': users,
        'count': len(users)
    }), 200

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """GET endpoint that returns a specific user by ID"""
    user = next((u for u in users if u['id'] == user_id), None)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': f'User with ID {user_id} not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'data': user
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask API is running'
    }), 200

if __name__ == '__main__':
    print('Flask API with user data endpoints created successfully!')
    print('Endpoints:')
    print('  GET /api/users - Returns all users')
    print('  GET /api/users/<id> - Returns specific user by ID')
    print('  GET /api/health - Health check')
    print('\nTo run the API, execute: flask run')
    print('Or: python flask_api.py')
