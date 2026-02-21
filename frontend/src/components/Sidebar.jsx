import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
    const location = useLocation();
    const menuItems = [
        { name: "Dashboard", path: "/" },
        { name: "Analysis History", path: "/history" },
        { name: "Settings", path: "/settings" }
    ];
    
    return (
        <div style={{ width: '260px', backgroundColor: '#1e293b', height: '100vh', padding: '25px 15px', borderRight: '1px solid #334155' }}>
            <div style={{ marginBottom: '40px', textAlign: 'center' }}>
                <div style={{ width: '60px', height: '60px', backgroundColor: '#3b82f6', borderRadius: '50%', margin: '0 auto 10px', boxShadow: '0 0 15px rgba(59, 130, 246, 0.5)' }}></div>
                <h4 style={{ margin: 0 }}>DeepGuard Admin</h4>
            </div>

            {menuItems.map(item => (
                <Link to={item.path} key={item.name} style={{ textDecoration: 'none' }}>
                    <div style={{ 
                        padding: '12px 20px', 
                        color: location.pathname === item.path ? '#38bdf8' : '#94a3b8', 
                        cursor: 'pointer', 
                        borderRadius: '10px', 
                        marginBottom: '10px', 
                        transition: '0.3s',
                        backgroundColor: location.pathname === item.path ? '#0f172a' : 'transparent',
                        fontWeight: location.pathname === item.path ? 'bold' : 'normal',
                        borderLeft: location.pathname === item.path ? '4px solid #38bdf8' : '4px solid transparent'
                    }}>
                        {item.name}
                    </div>
                </Link>
            ))}
        </div>
    );
};

export default Sidebar;