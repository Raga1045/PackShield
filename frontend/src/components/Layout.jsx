import React from "react";
import { NavLink } from "react-router-dom";
import "./Layout.css";
import { FaHome, FaBarcode, FaBoxes, FaBell, FaShieldAlt } from "react-icons/fa";

const Layout = ({ children }) => {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand-logo-container">
          <FaShieldAlt className="brand-icon-shield" />
          <span className="brand-name">PackShield</span>
        </div>

        <nav>
          <NavLink exact="true" to="/" activeclassname="active-link">
            <FaHome className="icon" />
            Home
          </NavLink>
          <NavLink to="/scan" activeclassname="active-link">
            <FaBarcode className="icon" />
            Scan
          </NavLink>
          <NavLink to="/inventory" activeclassname="active-link">
            <FaBoxes className="icon" />
            Inventory
          </NavLink>
          <NavLink to="/details" activeclassname="active-link">
            <FaBell className="icon" />
            Details
          </NavLink>
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
};

export default Layout;
