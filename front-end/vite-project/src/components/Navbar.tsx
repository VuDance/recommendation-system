import { Link, NavLink } from 'react-router-dom'
import './Navbar.css'

const Navbar = () => {
  const isLoggedIn = !!localStorage.getItem('token')

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('email')
    localStorage.removeItem('firstName')
    localStorage.removeItem('lastName')
    window.location.href = '/'
  }

  return (
    <nav className="navbar">
      <div className="navbar-container container">
        <div className="navbar-left">
          <Link to="/" className="navbar-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Recommendation System</span>
          </Link>
        </div>

        <div className="navbar-center">
          <NavLink to="/" className="navbar-link">Home</NavLink>
          <NavLink to="/recommendations" className="navbar-link">Recommendations</NavLink>
          <NavLink to="/trending" className="navbar-link">Trending</NavLink>
        </div>

        <div className="navbar-right">
          <button className="navbar-icon-button">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 19L13 13M15 8C15 11.866 11.866 15 8 15C4.13401 15 1 11.866 1 8C1 4.13401 4.13401 1 8 1C11.866 1 15 4.13401 15 8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button className="navbar-icon-button">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 3H5L5.4 5M7 13H13L16 5H5.4M7 13L5.4 5M7 13L4.70711 15.2929C4.07714 15.9229 4.52331 17 5.41421 17H15M15 17C14.4477 17 14 17.4477 14 18C14 18.5523 14.4477 19 15 19C15.5523 19 16 18.5523 16 18C16 17.4477 15.5523 17 15 17Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          {isLoggedIn ? (
            <button className="navbar-logout-btn" onClick={handleLogout}>
              Logout
            </button>
          ) : (
            <div className="navbar-auth-buttons">
              <Link to="/login" className="navbar-login-btn">
                Login
              </Link>
              <Link to="/register" className="navbar-register-btn">
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar