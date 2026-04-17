import api from './api'

interface JwtPayload {
  sub: string
  email: string
  iat: number
  exp: number
}

/**
 * Auth Service - Handles JWT token management and authentication
 */
export const authService = {
  /**
   * Get the stored JWT token
   */
  getToken: (): string | null => {
    return localStorage.getItem('token')
  },

  /**
   * Get the current user ID
   */
  getUserId: (): string | null => {
    return localStorage.getItem('userId')
  },

  /**
   * Get the current user info
   */
  getUserInfo: () => {
    return {
      userId: localStorage.getItem('userId'),
      email: localStorage.getItem('email'),
      firstName: localStorage.getItem('firstName'),
      lastName: localStorage.getItem('lastName'),
    }
  },

  /**
   * Extract user ID from JWT token
   * @param token JWT token
   * @returns User ID or null if invalid
   */
  extractUserIdFromToken: (token: string): string | null => {
    try {
      // JWT tokens have 3 parts separated by dots
      const parts = token.split('.')
      if (parts.length !== 3) return null

      // Decode the payload (second part)
      const payload = JSON.parse(atob(parts[1])) as JwtPayload
      return payload.sub || null
    } catch (error) {
      console.error('Error extracting user ID from token:', error)
      return null
    }
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated: (): boolean => {
    return !!localStorage.getItem('token')
  },

  /**
   * Logout user
   */
  logout: (): void => {
    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('email')
    localStorage.removeItem('firstName')
    localStorage.removeItem('lastName')
    delete api.defaults.headers.common['Authorization']
  },

  /**
   * Setup authorization header with token
   */
  setAuthHeader: (token: string): void => {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
  },
}

export default authService
