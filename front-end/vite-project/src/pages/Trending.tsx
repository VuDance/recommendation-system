import { useState, useEffect } from 'react'
import ProductCard from '../components/ProductCard'
import { getTrendingProducts } from '../services/api'
import './Trending.css'

const Trending = () => {
  const [trending, setTrending] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchTrending = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getTrendingProducts()
      setTrending(data)
    } catch (err) {
      setError('Failed to fetch trending products. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTrending()
  }, [])

  return (
    <div className="trending container">
      <div className="trending-header">
        <h1 className="page-title">Trending Products</h1>
        <p className="page-subtitle">
          Discover the most popular products across our platform based on view counts and user engagement.
        </p>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loading ? (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading trending products...</p>
        </div>
      ) : (
        <>
          {trending.length > 0 ? (
            <>
              <div className="trending-stats">
                <p>Showing {trending.length} trending products sorted by popularity</p>
              </div>
              <div className="trending-table">
                <table>
                  <thead>
                    <tr>
                      <th>Product</th>
                      <th>Views</th>
                      <th>Unique Users</th>
                      <th>Popularity Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trending.map((item, index) => (
                      <tr key={item.productId}>
                        <td>
                          <div className="product-cell">
                            <span className="trending-rank">{index + 1}</span>
                            <div className="product-info">
                              <span className="product-id">{item.productId}</span>
                              <span className="product-views">ASIN: {item.productId}</span>
                            </div>
                          </div>
                        </td>
                        <td>
                          <span className="view-count">{item.viewCount}</span>
                        </td>
                        <td>
                          <span className="user-count">{item.uniqueUsers}</span>
                        </td>
                        <td>
                          <div className="popularity-bar">
                            <div
                              className="popularity-fill"
                              style={{ width: `${Math.min(100, (item.viewCount / 100) * 100)}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="products-grid">
                <h2 className="section-title">Trending Products Grid</h2>
                <div className="grid">
                  {trending.map((item) => (
                    <ProductCard key={item.productId} product={item} />
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <p>No trending products found.</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default Trending