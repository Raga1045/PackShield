import axios from "axios";

const API_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const sendScanImage = async (imageBlob, metadata = {}) => {
  const formData = new FormData();
  formData.append("image", imageBlob);
  if (metadata.sku) formData.append("sku", metadata.sku);
  if (metadata.location) formData.append("location", metadata.location);

  const response = await axios.post(`${API_BASE}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

