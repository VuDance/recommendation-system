import { Link } from 'react-router-dom'
import './Home.css'

const Home = () => {
  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content container">
          <h1 className="hero-title">Personalized Product Recommendations</h1>
          <p className="hero-subtitle">
            Discover products tailored just for you with our advanced recommendation system.
          </p>
          <div className="hero-actions">
            <Link to="/recommendations" className="button button-primary">
              Get Recommendations
            </Link>
            <Link to="/trending" className="button button-secondary">
              View Trending
            </Link>
          </div>
        </div>
        <div className="hero-image"></div>
      </section>

      {/* Features Section */}
      <section className="features container">
        <h2 className="section-title">How It Works</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">1</div>
            <h3 className="feature-title">Personalized Analysis</h3>
            <p className="feature-description">
              Our system analyzes your browsing history and preferences to understand your unique taste.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">2</div>
            <h3 className="feature-title">Smart Recommendations</h3>
            <p className="feature-description">
              Get product suggestions that match your interests, powered by machine learning algorithms.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">3</div>
            <h3 className="feature-title">Trending Insights</h3>
            <p className="feature-description">
              Discover what's popular across our platform with real-time trending product analysis.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home