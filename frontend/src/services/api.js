import axios from "axios";

const API_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const fetchDashboardStats = () =>
  axios.get(`${API_BASE}/dashboard`);

export const fetchDetails = () =>
  axios.get(`${API_BASE}/details`);
