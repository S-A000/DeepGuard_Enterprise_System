import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const UserLogin = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.post('http://127.0.0.1:8000/api/login', { email, password });
            // Governance Logic: Session save karo
            localStorage.setItem('userRole', res.data.role);
            localStorage.setItem('userName', res.data.full_name);
            localStorage.setItem('userID', res.data.user_id);
            navigate('/dashboard');
        } catch (err) {
            alert(err.response?.data?.detail || "Login Failed!");
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.card}>
                <h2 style={{color: '#38bdf8'}}>👤 User Login</h2>
                <form onSubmit={handleLogin}>
                    <input style={styles.input} type="email" placeholder="Email" onChange={e => setEmail(e.target.value)} required />
                    <input style={styles.input} type="password" placeholder="Password" onChange={e => setPassword(e.target.value)} required />
                    <button type="submit" style={styles.btn}>Access System</button>
                </form>
                <p style={{marginTop: '15px'}}>New here? <span onClick={() => navigate('/register')} style={{color: '#38bdf8', cursor: 'pointer'}}>Register</span></p>
            </div>
        </div>
    );
};
// Styles (Wahi purane Register wale use karlein)
const styles = {
    container: { height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#020617' },
    card: { backgroundColor: '#1e293b', padding: '40px', borderRadius: '20px', textAlign: 'center', width: '380px', border: '1px solid #38bdf8' },
    input: { width: '100%', padding: '12px', marginBottom: '15px', borderRadius: '8px', border: '1px solid #334155', backgroundColor: '#0f172a', color: 'white', boxSizing: 'border-box' },
    btn: { width: '100%', padding: '12px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }
};
export default UserLogin;