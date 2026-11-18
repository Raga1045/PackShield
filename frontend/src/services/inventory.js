import axios from "axios";

const API_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const fetchInventoryList = async (limit = 100) => {
  const response = await axios.get(`${API_BASE}/history?limit=${limit}`);
  return response.data;
};

export const fetchDashboardOverview = async () => {
  const response = await axios.get(`${API_BASE}/stats`);
  return response.data;
};

