import api from "./api"

export const getProducts = async (page: number =1, size: number =10) => {
  try {
    const response = await api.get("/products", { params: { page, size } })
    return response.data
  } catch (error) {
    console.error('Error fetching products:', error)
    throw error
  }
}

export const getProductById = async (productId: string) => {
  try {
    const response = await api.get(`/products/${productId}`)
    return response.data
    } catch (error) {
    console.error(`Error fetching product with ID ${productId}:`, error)
    throw error
  }
}