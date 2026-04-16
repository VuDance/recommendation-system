import { useState, useEffect } from 'react'
import ProductCard from '../components/ProductCard'
import './Recommendations.css'
import { getProducts } from '../services/product-service'

const Recommendations = () => {
  const [userId, setUserId] = useState('user123')
  const [recommendations, setRecommendations] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchRecommendations = async () => {
    if (!userId.trim()) return

    setLoading(true)
    setError(null)
    try {
      const data = await getProducts(1, 10) // Fetch first page of products as recommendations
      setRecommendations(data.content)
    } catch (err) {
      setError('Failed to fetch recommendations. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRecommendations()
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    fetchRecommendations()
  }

  return (
    <div className="recommendations container">
      <div className="recommendations-header">
        <h1 className="page-title">Personalized Recommendations</h1>
        <p className="page-subtitle">
          Enter a user ID to get personalized product recommendations based on their preferences.
        </p>

        <form onSubmit={handleSubmit} className="user-id-form">
          <div className="form-group">
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter user ID"
              className="form-input"
            />
            <button type="submit" className="button button-primary" disabled={loading}>
              {loading ? 'Loading...' : 'Get Recommendations'}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loading ? (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading recommendations...</p>
        </div>
      ) : (
        <>
          {recommendations.length > 0 ? (
            <>
              <div className="recommendations-stats">
                <p>Found {recommendations.length} recommendations for user <strong>{userId}</strong></p>
              </div>
              <div className="products-grid">
                {recommendations.map((product) => (
                  <ProductCard key={product.id || product.productId} product={product} />
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <p>No recommendations found. Try a different user ID.</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default Recommendations