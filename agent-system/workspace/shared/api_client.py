import requests
import json
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin

class APIClient:
    """Client for fetching user data from REST APIs"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self._setup_headers()
    
    def _setup_headers(self) -> None:
        """Setup default headers for requests"""
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments to pass to requests
        
        Returns:
            Response JSON as dictionary
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.text else {}
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def get_user(self, user_id: int) -> Dict[str, Any]:
        """
        Fetch a single user by ID
        
        Args:
            user_id: User ID
        
        Returns:
            User data dictionary
        """
        return self._make_request('GET', f'/users/{user_id}')
    
    def get_users(self, page: int = 1, limit: int = 10, **filters) -> List[Dict[str, Any]]:
        """
        Fetch multiple users with pagination and filtering
        
        Args:
            page: Page number (default: 1)
            limit: Items per page (default: 10)
            **filters: Additional filter parameters
        
        Returns:
            List of user data dictionaries
        """
        params = {'page': page, 'limit': limit}
        params.update(filters)
        response = self._make_request('GET', '/users', params=params)
        return response if isinstance(response, list) else response.get('data', [])
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user by email address
        
        Args:
            email: User email address
        
        Returns:
            User data dictionary or None if not found
        """
        try:
            response = self._make_request('GET', '/users', params={'email': email})
            users = response if isinstance(response, list) else response.get('data', [])
            return users[0] if users else None
        except Exception:
            return None
    
    def create_user(self, name: str, email: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new user
        
        Args:
            name: User name
            email: User email
            **kwargs: Additional user fields
        
        Returns:
            Created user data
        """
        data = {'name': name, 'email': email}
        data.update(kwargs)
        return self._make_request('POST', '/users', json=data)
    
    def update_user(self, user_id: int, **kwargs) -> Dict[str, Any]:
        """
        Update an existing user
        
        Args:
            user_id: User ID
            **kwargs: Fields to update
        
        Returns:
            Updated user data
        """
        return self._make_request('PUT', f'/users/{user_id}', json=kwargs)
    
    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user
        
        Args:
            user_id: User ID
        
        Returns:
            True if deletion was successful
        """
        try:
            self._make_request('DELETE', f'/users/{user_id}')
            return True
        except Exception:
            return False
    
    def search_users(self, query: str, **filters) -> List[Dict[str, Any]]:
        """
        Search users by query
        
        Args:
            query: Search query string
            **filters: Additional filter parameters
        
        Returns:
            List of matching users
        """
        params = {'q': query}
        params.update(filters)
        response = self._make_request('GET', '/users/search', params=params)
        return response if isinstance(response, list) else response.get('data', [])
    
    def close(self) -> None:
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage
if __name__ == '__main__':
    # Initialize client
    client = APIClient(
        base_url='https://jsonplaceholder.typicode.com',
        timeout=10
    )
    
    try:
        # Fetch single user
        print('=== Fetch User 1 ===')
        user = client.get_user(1)
        print(json.dumps(user, indent=2))
        
        # Fetch multiple users
        print('\n=== Fetch Users (paginated) ===')
        users = client.get_users(page=1, limit=3)
        print(f'Found {len(users)} users')
        for u in users[:2]:
            print(f"  - {u.get('name', 'N/A')} ({u.get('email', 'N/A')})")
        
        # Create user
        print('\n=== Create User ===')
        new_user = client.create_user(
            name='Test User',
            email='test@example.com',
            phone='555-1234'
        )
        print(json.dumps(new_user, indent=2))
        
        # Update user
        print('\n=== Update User ===')
        updated = client.update_user(1, name='Updated Name')
        print(json.dumps(updated, indent=2))
        
    except Exception as e:
        print(f'Error: {e}')
    finally:
        client.close()
