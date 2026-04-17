import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import api from '../services/api'
import './ProductDetail.css'

interface Recommendation {
  id: number
  title: string
  brand: string
  imageURL: string[]
}

const ProductDetail = () => {
  const { productId } = useParams()
  const [product, setProduct] = useState<Product | null>(null)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchProductDetail = async () => {
      try {
        setLoading(true)
        const response = await api.get(`/products/${productId}`)
        setProduct(response.data)
      } catch (err: any) {
        setError('Failed to fetch product details')
      }
    }

    const fetchRecommendations = async () => {
      try {
        const userId = localStorage.getItem('userId')
        const response = await api.get(`/recommendations/${userId}`)
        setRecommendations(response.data)
      } catch (err: any) {
        console.log('Failed to fetch recommendations')
      }
    }

    if (productId) {
      fetchProductDetail()
      fetchRecommendations()
    }

    setLoading(false)
  }, [productId])

  if (loading) {
    return <div className="loading">Loading...</div>
  }

  if (error) {
    return <div className="error">{error}</div>
  }

  if (!product) {
    return <div className="error">Product not found</div>
  }

  return (
    <div className="product-detail-container">
      <div className="product-detail">
        <div className="product-images">
          {product.imageURL && product.imageURL.length > 0 ? (
            <img src={product.imageURL[0]} alt={product.title} />
          ) : (
            <div className="no-image">No image available</div>
          )}
        </div>

        <div className="product-info">
          <h1>{product.title}</h1>
          <p className="brand">Brand: {product.brand || 'N/A'}</p>
          <p className="description">{product.description}</p>
          <button className="add-to-cart-btn">Add to Cart</button>
        </div>
      </div>

      {/* Recommendations Section */}
      <div className="recommendations-section">
        <h2>Recommended for You</h2>
        {recommendations.length > 0 ? (
          <div className="recommendations-grid">
            {recommendations.map((rec) => (
              <div key={rec.id} className="recommendation-card">
                <div className="rec-image">
                  {rec.imageURL && rec.imageURL.length > 0 ? (
                    <img src={rec.imageURL[0]} alt={rec.title} />
                  ) : (
                    <div className="no-image">No image</div>
                  )}
                </div>
                <div className="rec-info">
                  <h3>{rec.title}</h3>
                  <p>{rec.brand || 'N/A'}</p>
                  <a href={`/product/${rec.id}`} className="view-btn">
                    View Details
                  </a>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="no-recommendations">No recommendations available at this time.</p>
        )}
      </div>
    </div>
  )
}

export default ProductDetail
