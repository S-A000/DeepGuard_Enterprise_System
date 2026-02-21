import React from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
    const navigate = useNavigate();

    const handleLoginEntry = (role) => {
        if (role === 'admin') {
            navigate('/login-admin');
        } else {
            navigate('/login-user');
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.overlay}></div>

            <div style={styles.contentBox}>
                <h1 style={styles.title}>
                    🛡️ DeepGuard <span style={styles.version}>Enterprise</span>
                </h1>
                <p style={styles.subtitle}>AI-Powered Video Forensics & Threat Intelligence</p>

                <div style={styles.divider}></div>

                <h3 style={{ color: '#cbd5e1', marginBottom: '30px', fontWeight: '400', letterSpacing: '1px' }}>
                    SECURE GATEWAY: SELECT ACCESS LEVEL
                </h3>

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

            <style>{`
                @keyframes fadeIn { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
                @keyframes glow { 0% { text-shadow: 0 0 10px #38bdf8; } 50% { text-shadow: 0 0 25px #38bdf8, 0 0 40px #2563eb; } 100% { text-shadow: 0 0 10px #38bdf8; } }
                .login-btn { transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }
                .login-btn:hover { transform: translateY(-10px); cursor: pointer; }
                .admin-btn:hover { border-color: #ef4444 !important; background-color: rgba(127, 29, 29, 0.2) !important; box-shadow: 0 15px 30px rgba(239, 68, 68, 0.2); }
                .user-btn:hover { border-color: #38bdf8 !important; background-color: rgba(3, 105, 161, 0.2) !important; box-shadow: 0 15px 30px rgba(56, 189, 248, 0.2); }
            `}</style>
        </div>
    );
};

const styles = {
    container: { height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#020617', position: 'relative', overflow: 'hidden', color: 'white', fontFamily: "'Inter', sans-serif" },
    overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundImage: 'radial-gradient(circle at center, #1e293b 0%, #020617 100%)', pointerEvents: 'none' },
    contentBox: { textAlign: 'center', zIndex: 2, animation: 'fadeIn 1s ease-out', padding: '60px', borderRadius: '32px', backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255, 255, 255, 0.1)', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' },
    title: { fontSize: '4.5rem', margin: '0', color: 'white', fontWeight: '900', letterSpacing: '-2px', animation: 'glow 3s infinite ease-in-out' },
    version: { fontSize: '1.2rem', color: '#38bdf8', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '3px', marginLeft: '10px' },
    subtitle: { color: '#94a3b8', fontSize: '1.1rem', marginTop: '10px', fontWeight: '300' },
    divider: { height: '1px', width: '100px', background: '#334155', margin: '40px auto' },
    buttonContainer: { display: 'flex', gap: '25px', justifyContent: 'center', marginTop: '20px' },
    adminButton: { padding: '40px 30px', borderRadius: '24px', backgroundColor: 'transparent', color: '#fca5a5', width: '250px' },
    userButton: { padding: '40px 30px', borderRadius: '24px', backgroundColor: 'transparent', color: '#7dd3fc', width: '250px' }
};

export default LandingPage;