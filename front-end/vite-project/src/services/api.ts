import axios from 'axios'

// Create axios instance with base URL
const api = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL}/api`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token if needed
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for handling common errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error:', error.response.status, error.response.data)

      if (error.response.status === 401) {
        // Handle unauthorized access
        console.log('Unauthorized access')
      } else if (error.response.status === 404) {
        // Handle not found
        console.log('Resource not found')
      } else if (error.response.status >= 500) {
        // Handle server errors
        console.log('Server error occurred')
      }
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request)
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Request setup error:', error.message)
    }

    return Promise.reject(error)
  }
)

// API functions
export const getRecommendations = async (userId: string) => {
  try {
    const response = await api.get(`/recommendations/${userId}`)
    return response.data
  } catch (error) {
    console.error('Error fetching recommendations:', error)
    throw error
  }
}

export const getTrendingProducts = async () => {
  try {
    const response = await api.get('/recommendations/trending')
    return response.data
  } catch (error) {
    console.error('Error fetching trending products:', error)
    throw error
  }
}

// Additional product-related APIs can be added here as needed
export default api