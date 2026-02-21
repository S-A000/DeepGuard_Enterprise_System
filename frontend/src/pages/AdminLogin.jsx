import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AdminLogin = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        // Admin Credentials: email: admin@deepguard.com | pass: admin123
        if (email === 'admin@deepguard.com' && password === 'admin123') {
            localStorage.setItem('userRole', 'admin');
            localStorage.setItem('userEmail', email);
            navigate('/dashboard');
        } else {
            alert("⚠️ ACCESS DENIED: Invalid Admin Credentials");
        }
    };

    return (
        <div style={styles.fullPage}>
            <div style={styles.loginCard}>
                <div style={styles.iconCircle}>🛡️</div>
                <h2 style={{ color: '#ef4444', marginBottom: '10px' }}>Admin Command Center</h2>
                <p style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '25px' }}>High-Level Authorization Required</p>
                
                <form onSubmit={handleLogin}>
                    <input 
                        type="email" 
                        placeholder="Admin Email" 
                        style={styles.input}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                    <input 
                        type="password" 
                        placeholder="Security Key" 
                        style={styles.input}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <button type="submit" style={styles.adminBtn}>Verify & Authorize</button>
                </form>
                
                <p onClick={() => navigate('/')} style={styles.backLink}>← Return to Gateway</p>
            </div>
        </div>
    );
};

// --- STYLES OBJECT (Iska hona zaroori hai) ---
const styles = {
    fullPage: {
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#020617',
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    },
    loginCard: {
        backgroundColor: '#1e293b',
        padding: '40px',
        borderRadius: '20px',
        textAlign: 'center',
        width: '380px',
        border: '1px solid #ef4444',
        boxShadow: '0 0 30px rgba(239, 68, 68, 0.2)'
    },
    iconCircle: {
        fontSize: '40px',
        marginBottom: '15px'
    },
    input: {
        width: '100%',
        padding: '12px',
        marginBottom: '15px',
        borderRadius: '8px',
        border: '1px solid #334155',
        backgroundColor: '#0f172a',
        color: 'white',
        outline: 'none',
        boxSizing: 'border-box'
    },
    adminBtn: {
        width: '100%',
        padding: '12px',
        backgroundColor: '#ef4444',
        color: 'white',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        fontWeight: 'bold',
        fontSize: '16px',
        marginTop: '10px',
        transition: '0.3s'
    },
    backLink: {
        color: '#64748b',
        marginTop: '25px',
        cursor: 'pointer',
        fontSize: '14px',
        display: 'block',
        textDecoration: 'underline'
    }
};

export default AdminLogin;