import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const Sidebar = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const userRole = localStorage.getItem('role');

    const handleLogout = () => {
        localStorage.clear();
        navigate('/');
    };

    return (
        <div style={styles.sidebar}>
            <div style={styles.topSection}>
                <div style={styles.logoBox}>
                    <div style={styles.icon}>🏛️</div>
                    <h3 style={styles.logoText}>DeepGuard</h3>
                    <span style={styles.roleTag}>{userRole?.toUpperCase()}</span>
                </div>

                <div style={styles.menu}>
                    <p style={styles.label}>CORE WORKSPACE</p>
                    <Link to="/dashboard" style={styles.link(location.pathname === '/dashboard')}>📊 Dashboard</Link>
                    <Link to="/history" style={styles.link(location.pathname === '/history')}>📜 Forensic History</Link>

                    {userRole === 'admin' && (
                        <>
                            <p style={styles.label}>IDENTITY MANAGEMENT</p>
                            <Link to="/assign-member" style={styles.link(location.pathname === '/assign-member')}>➕ Assign New Member</Link>
                            <Link to="/user-management" style={styles.link(location.pathname === '/user-management')}>👥 User Management</Link>
                        </>
                    )}
                </div>
            </div>

            <div style={styles.sidebarFooter}>
                <button onClick={handleLogout} style={styles.logoutBtn}>
                    🚪 EXIT SYSTEM
                </button>
            </div>
        </div>
    );
};

const styles = {
    sidebar: { width: '260px', backgroundColor: '#0f172a', height: '100vh', position: 'fixed', left: 0, top: 0, display: 'flex', flexDirection: 'column', borderRight: '1px solid #1e293b', zIndex: 100 },
    topSection: { flex: 1, padding: '30px 20px', overflowY: 'auto' },
    logoBox: { textAlign: 'center', marginBottom: '40px' },
    icon: { fontSize: '40px', marginBottom: '10px' },
    logoText: { margin: 0, color: '#38bdf8', letterSpacing: '1px' },
    roleTag: { fontSize: '10px', color: '#64748b', fontWeight: 'bold', letterSpacing: '2px' },
    label: { fontSize: '11px', color: '#475569', letterSpacing: '1px', margin: '25px 0 15px 10px' },
    link: (active) => ({ display: 'block', padding: '12px 15px', borderRadius: '12px', textDecoration: 'none', color: active ? '#38bdf8' : '#94a3b8', backgroundColor: active ? '#1e293b' : 'transparent', marginBottom: '8px', fontWeight: active ? 'bold' : 'normal', transition: '0.3s' }),
    sidebarFooter: { padding: '20px', borderTop: '1px solid #1e293b', backgroundColor: '#0f172a' },
    logoutBtn: { width: '100%', padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid #ef4444', borderRadius: '10px', cursor: 'pointer', fontWeight: 'bold', letterSpacing: '1px', transition: '0.3s' }
};

export default Sidebar;