import axios from "axios";

const API_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const getDashboardStats = async () => {
  try {
    const response = await axios.get(`${API_URL}/dashboard`);
    return response.data;
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    return null;
  }
};
