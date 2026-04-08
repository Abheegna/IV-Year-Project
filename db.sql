DROP DATABASE IF EXISTS stress_detection;

CREATE DATABASE stress_detection;
USE stress_detection;

CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(100),
  email VARCHAR(100) UNIQUE,
  phone VARCHAR(20),
  password VARCHAR(100)
);