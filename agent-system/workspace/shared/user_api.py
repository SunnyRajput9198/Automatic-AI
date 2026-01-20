from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from typing import List, Dict, Optional

app = Flask(__name__)
CORS(app)

# Mock user database
users_db = [
    {'id': 1, 'name': 'Alice Johnson', 'email': 'alice@example.com', 'created_at': '2023-01-15'},
    {'id': 2, 'name': 'Bob Smith', 'email': 'bob@example.com', 'created_at': '2023-02-20'},
    {'id': 3, 'name': 'Carol White', 'email': 'carol@example.com', 'created_at': '2023-03-10'},
]

# GET all users
@app.route('/api/users', methods=['GET'])
def get_users():
    """Retrieve all users from the database"""
    return jsonify({
        'status': 'success',
        'data': users_db,
        'count': len(users_db)
    }), 200

# GET single user by ID
@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Retrieve a specific user by ID"""
    user = next((u for u in users_db if u['id'] == user_id), None)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'User not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'data': user
    }), 200

# POST create new user
@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    data = request.get_json()
    
    # Validation
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({
            'status': 'error',
            'message': 'Name and email are required'
        }), 400
    
    new_user = {
        'id': max([u['id'] for u in users_db]) + 1,
        'name': data['name'],
        'email': data['email'],
        'created_at': datetime.now().strftime('%Y-%m-%d')
    }
    
    users_db.append(new_user)
    
    return jsonify({
        'status': 'success',
        'message': 'User created successfully',
        'data': new_user
    }), 201

# PUT update user
@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update an existing user"""
    user = next((u for u in users_db if u['id'] == user_id), None)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'User not found'
        }), 404
    
    data = request.get_json()
    user.update({k: v for k, v in data.items() if k in ['name', 'email']})
    
    return jsonify({
        'status': 'success',
        'message': 'User updated successfully',
        'data': user
    }), 200

# DELETE user
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    global users_db
    user = next((u for u in users_db if u['id'] == user_id), None)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'User not found'
        }), 404
    
    users_db = [u for u in users_db if u['id'] != user_id]
    
    return jsonify({
        'status': 'success',
        'message': 'User deleted successfully'
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
