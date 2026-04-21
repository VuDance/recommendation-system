import { Link } from 'react-router-dom'
import './ProductCard.css'

interface Product {
  id: number
  asin: string
  title: string
  description: string
  brand: string
  imageURL: string[]
}

interface ProductCardProps {
    product: Product
}

const ProductCard = ({ product }: ProductCardProps) => {
  // Extract product information based on possible data structures
  const productId = product.asin
  const title = product.title
  const price =  '$99.99'
  const brand = product.brand || 'Brand'
  const imageUrl = product.imageURL && product.imageURL.length > 0 ? product.imageURL : [`https://axiomprint.com/icons/default-squre.jpg`]
  const description = product.description || 'No description available.'

  return (
    <Link to={`/product/${productId}`} style={{ textDecoration: 'none' }}>
      <div className="product-card">
        <div className="product-image">
          <img src={imageUrl[0]} alt={title} loading="lazy" />
        </div>
        <div className="product-info">
          <div className="product-header">
            <span className="product-brand">{brand}</span>
            <h3 className="product-title">{title}</h3>
          </div>
          {description && (
            <p className="product-description">{description}</p>
          )}
          <div className="product-footer">
            <span className="product-price">{price}</span>
            <button className="product-button">
              View Details
            </button>
          </div>
        </div>
      </div>
    </Link>
  )
}

export default ProductCard