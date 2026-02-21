import React from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
    const navigate = useNavigate();

    const handleLoginEntry = (role) => {
        // Role ke hisaab se sahi login page par bhejna
        if (role === 'admin') {
            navigate('/login-admin');
        } else {
            navigate('/login-user');
        }
    };

    return (
        <div style={styles.container}>
            {/* Cyberpunk Grid Background */}
            <div style={styles.overlay}></div>

            <div style={styles.contentBox}>
                {/* Title Section with Glow Animation */}
                <h1 style={styles.title}>
                    🛡️ DeepGuard <span style={styles.version}>Enterprise</span>
                </h1>
                <p style={styles.subtitle}>AI-Powered Video Forensics & Threat Intelligence</p>

                <div style={styles.divider}></div>

                <h3 style={{ color: '#cbd5e1', marginBottom: '30px', fontWeight: '400', letterSpacing: '1px' }}>
                    SECURE GATEWAY: SELECT ACCESS LEVEL
                </h3>

                {/* Login Options Buttons */}
                <div style={styles.buttonContainer}>
                    
                    {/* Admin Access Card */}
                    <button 
                        style={styles.adminButton} 
                        className="login-btn admin-btn"
                        onClick={() => handleLoginEntry('admin')}
                    >
                        <div style={{ fontSize: '35px', marginBottom: '10px' }}>👨‍💻</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>ADMIN ACCESS</div>
                        <small style={{ color: '#94a3b8', fontSize: '11px', display: 'block', marginTop: '5px' }}>
                            Full System Control & Logs
                        </small>
                    </button>

                    {/* User Access Card */}
                    <button 
                        style={styles.userButton} 
                        className="login-btn user-btn"
                        onClick={() => handleLoginEntry('user')}
                    >
                        <div style={{ fontSize: '35px', marginBottom: '10px' }}>👤</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>USER LOGIN</div>
                        <small style={{ color: '#94a3b8', fontSize: '11px', display: 'block', marginTop: '5px' }}>
                            Video Analysis & Reports
                        </small>
                    </button>
                    
                </div>
            </div>

            {/* Stylish CSS Animations */}
            <style>{`
                @keyframes fadeIn { 
                    from { opacity: 0; transform: translateY(30px); } 
                    to { opacity: 1; transform: translateY(0); } 
                }
                @keyframes glow { 
                    0% { text-shadow: 0 0 10px #38bdf8; } 
                    50% { text-shadow: 0 0 25px #38bdf8, 0 0 40px #2563eb; } 
                    100% { text-shadow: 0 0 10px #38bdf8; } 
                }
                
                .login-btn { transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative; overflow: hidden; }
                .login-btn:hover { transform: translateY(-10px); cursor: pointer; }
                
                .admin-btn:hover { 
                    box-shadow: 0 15px 35px -5px rgba(239, 68, 68, 0.4); 
                    border-color: #ef4444 !important; 
                    background-color: rgba(127, 29, 29, 0.3) !important;
                }
                .user-btn:hover { 
                    box-shadow: 0 15px 35px -5px rgba(56, 189, 248, 0.4); 
                    border-color: #38bdf8 !important; 
                    background-color: rgba(3, 105, 161, 0.3) !important;
                }

                /* Mobile responsiveness */
                @media (max-width: 600px) {
                    .buttonContainer { flex-direction: column; }
                }
            `}</style>
        </div>
    );
};

// Styles Object for High-End Look
const styles = {
    container: {
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#020617',
        backgroundImage: 'radial-gradient(circle at center, #1e293b 0%, #020617 100%)',
        position: 'relative',
        overflow: 'hidden',
        color: 'white',
        fontFamily: "'Inter', 'Segoe UI', sans-serif"
    },
    overlay: {
        position: 'absolute',
        top: 0, left: 0, right: 0, bottom: 0,
        backgroundImage: 'linear-gradient(rgba(56, 189, 248, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(56, 189, 248, 0.05) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
        pointerEvents: 'none'
    },
    contentBox: {
        textAlign: 'center',
        zIndex: 2,
        animation: 'fadeIn 1s ease-out',
        padding: '50px',
        borderRadius: '24px',
        backgroundColor: 'rgba(15, 23, 42, 0.7)',
        backdropFilter: 'blur(15px)',
        border: '1px solid rgba(56, 189, 248, 0.2)',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)'
    },
    title: {
        fontSize: '4.5rem',
        margin: '0',
        color: 'white',
        fontWeight: '900',
        letterSpacing: '-3px',
        animation: 'glow 3s infinite ease-in-out'
    },
    version: {
        fontSize: '1.2rem',
        color: '#38bdf8',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        letterSpacing: '3px',
        marginLeft: '10px'
    },
    subtitle: {
        color: '#94a3b8',
        fontSize: '1.1rem',
        marginTop: '10px',
        fontWeight: '300'
    },
    divider: {
        height: '1px',
        width: '150px',
        background: 'linear-gradient(90deg, transparent, #38bdf8, transparent)',
        margin: '40px auto'
    },
    buttonContainer: {
        display: 'flex',
        gap: '25px',
        justifyContent: 'center',
        marginTop: '20px'
    },
    adminButton: {
        padding: '35px 25px',
        borderRadius: '20px',
        border: '1px solid #7f1d1d',
        backgroundColor: 'rgba(127, 29, 29, 0.1)',
        color: '#fca5a5',
        width: '240px',
        cursor: 'pointer'
    },
    userButton: {
        padding: '35px 25px',
        borderRadius: '20px',
        border: '1px solid #0369a1',
        backgroundColor: 'rgba(3, 105, 161, 0.1)',
        color: '#7dd3fc',
        width: '240px',
        cursor: 'pointer'
    }
};

export default LandingPage;