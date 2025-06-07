// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAsLFrKVxNwsHuSmbfWMBjVFcn0WkXoWGs",
  authDomain: "waitless-aca8b.firebaseapp.com",
  projectId: "waitless-aca8b",
  storageBucket: "waitless-aca8b.firebasestorage.app",
  messagingSenderId: "313422936197",
  appId: "1:313422936197:web:9c0d0d3b4410b4d98cd1ec",
  measurementId: "G-0WH6DMQQJM"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const db = getFirestore(app);