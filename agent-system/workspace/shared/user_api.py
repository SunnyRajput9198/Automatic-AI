from flask import Flask, jsonify, request
import requests
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

load_dotenv()
app = Flask(__name__)

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.example.com')
API_KEY = os.getenv('API_KEY')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Fetch a single user by ID"""
    try:
        headers = {'Authorization': f'Bearer {API_KEY}'}
        response = requests.get(
            f'{API_BASE_URL}/users/{user_id}',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        logger.error(f'Error fetching user {user_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Fetch all users with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        headers = {'Authorization': f'Bearer {API_KEY}'}
        params = {'page': page, 'limit': limit}
        
        response = requests.get(
            f'{API_BASE_URL}/users',
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        logger.error(f'Error fetching all users: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{API_BASE_URL}/users',
            json=data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        logger.info(f'User created successfully')
        return jsonify(response.json()), 201
    except requests.exceptions.RequestException as e:
        logger.error(f'Error creating user: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update an existing user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.put(
            f'{API_BASE_URL}/users/{user_id}',
            json=data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        logger.info(f'User {user_id} updated successfully')
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        logger.error(f'Error updating user {user_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        headers = {'Authorization': f'Bearer {API_KEY}'}
        response = requests.delete(
            f'{API_BASE_URL}/users/{user_id}',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        logger.info(f'User {user_id} deleted successfully')
        return jsonify({'message': 'User deleted successfully'}), 200
    except requests.exceptions.RequestException as e:
        logger.error(f'Error deleting user {user_id}: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f'Internal server error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)