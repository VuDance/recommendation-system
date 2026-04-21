import { useState, useEffect } from 'react'
import ProductCard from '../components/ProductCard'
import './Products.css'
import { getProducts } from '../services/product-service'

const Products = () => {
  const [page, setPage] = useState(1)
  const [products, setProducts] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchProducts = async (page: number) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getProducts(page, 10)
      setProducts((prevProducts) => [...prevProducts, ...data.content])
    } catch (err) {
      setError('Failed to fetch products. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadMore = () => {
    setPage((prevPage) => prevPage + 1)
  }

  useEffect(() => {
    fetchProducts(page)
  }, [page])

  return (
    <div className="products container">
          <div className="products-grid">
                {products.map((product) => (
                  <ProductCard key={product.id || product.productId} product={product} />
                ))}
              </div>
          <div className='load-more'>
              {loading ? (
                <div className="loading">
                  <div className="loading-spinner"></div>
                  <p>Loading...</p>
                </div>
              ) : (
                <>
                <button className='btn' onClick={handleLoadMore}>
                  Load more
                </button>
                </>
              )}
          </div>
      
    </div>
  )
}

export default Products