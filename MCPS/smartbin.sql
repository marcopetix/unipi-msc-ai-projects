-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Creato il: Mag 18, 2020 alle 18:28
-- Versione del server: 10.4.11-MariaDB
-- Versione PHP: 7.4.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `smartbin`
--

-- --------------------------------------------------------

--
-- Struttura della tabella `bin`
--

CREATE TABLE `bin` (
  `id` int(11) NOT NULL,
  `type` set('paper','plastic','organic','glass','unsorted') NOT NULL,
  `capacity` int(3) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dump dei dati per la tabella `bin`
--

INSERT INTO `bin` (`id`, `type`, `capacity`) VALUES
(1, 'paper', 100),
(2, 'plastic', 0),
(3, 'organic', 0),
(4, 'glass', 0),
(5, 'unsorted', 0);

-- --------------------------------------------------------

--
-- Struttura della tabella `user`
--

CREATE TABLE `user` (
  `card_id` varchar(30) NOT NULL,
  `name` varchar(50) NOT NULL,
  `sorted_wastes` int(11) NOT NULL DEFAULT 0,
  `unsorted_wastes` int(11) NOT NULL DEFAULT 0,
  `points` int(11) NOT NULL DEFAULT 0,
  `isAdmin` bit(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dump dei dati per la tabella `user`
--

INSERT INTO `user` (`card_id`, `name`, `sorted_wastes`, `unsorted_wastes`, `points`, `isAdmin`) VALUES
('_07_B1_89_44', 'Dawit', 100, 11, 89, b'0'),
('_49_4B_F9_D5', 'Marco', 30, 14, 16, b'0'),
('_99_F0_FD_D5', 'Alessio ', 42, 0, 42, b'1'),
('_C9_B9_9C_96', 'Diletta', 60, 0, 60, b'0');

--
-- Trigger `user`
--
DELIMITER $$
CREATE TRIGGER `update_points` BEFORE UPDATE ON `user` FOR EACH ROW BEGIN
    SET NEW.points = NEW.sorted_wastes - NEW.unsorted_wastes;
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Struttura della tabella `user_interaction`
--

CREATE TABLE `user_interaction` (
  `id` int(11) NOT NULL,
  `user_id` varchar(30) NOT NULL,
  `bin_id` int(11) NOT NULL,
  `garbage_weight` int(11) DEFAULT NULL,
  `timestamp` date NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dump dei dati per la tabella `user_interaction`
--

INSERT INTO `user_interaction` (`id`, `user_id`, `bin_id`, `garbage_weight`, `timestamp`) VALUES
(5, '_49_4B_F9_D5', 1, 100, '2020-05-16'),
(7, '_49_4B_F9_D5', 1, 62, '2020-05-16'),
(8, '_49_4B_F9_D5', 1, 0, '2020-05-16'),
(9, '_99_F0_FD_D5', 1, 40, '2020-05-16'),
(10, '_49_4B_F9_D5', 1, 23, '2020-05-16'),
(11, '_49_4B_F9_D5', 1, 68, '2020-05-16'),
(12, '_99_F0_FD_D5', 1, 0, '2020-05-16'),
(13, '_99_F0_FD_D5', 1, 0, '2020-05-17'),
(14, '_49_4B_F9_D5', 1, 18831, '2020-05-18'),
(15, '_49_4B_F9_D5', 1, 17815, '2020-05-18'),
(16, '_07_B1_89_44', 1, 0, '2020-05-18'),
(17, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(18, '_99_F0_FD_D5', 1, 160265, '2020-05-18'),
(19, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(20, '_99_F0_FD_D5', 1, 82527, '2020-05-18'),
(21, '_49_4B_F9_D5', 1, 18066, '2020-05-18'),
(22, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(23, '_49_4B_F9_D5', 1, 0, '2020-05-18'),
(24, '_C9_B9_9C_96', 1, 0, '2020-05-18'),
(25, '_C9_B9_9C_96', 1, 0, '2020-05-18'),
(26, '_C9_B9_9C_96', 1, 83549, '2020-05-18'),
(27, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(28, '_07_B1_89_44', 1, 84684, '2020-05-18'),
(29, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(30, '_07_B1_89_44', 1, 84584, '2020-05-18'),
(31, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(32, '_07_B1_89_44', 1, 92585, '2020-05-18'),
(33, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(34, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(35, '_99_F0_FD_D5', 1, 84675, '2020-05-18'),
(36, '_07_B1_89_44', 1, 102, '2020-05-18'),
(37, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(38, '_99_F0_FD_D5', 1, 0, '2020-05-18'),
(39, '_07_B1_89_44', 1, 106, '2020-05-18');

--
-- Trigger `user_interaction`
--
DELIMITER $$
CREATE TRIGGER `user_garbage_update` AFTER INSERT ON `user_interaction` FOR EACH ROW BEGIN
    IF NEW.garbage_weight IS NOT NULL THEN
    	SELECT type INTO @garbage_type FROM bin WHERE id = NEW.bin_id;
        IF @garbage_type != 'unsorted' THEN
        	SELECT sorted_wastes INTO @previous_weight FROM user WHERE card_id = NEW.user_id;
            UPDATE user SET sorted_wastes = @previous_weight + NEW.garbage_weight WHERE card_id = NEW.user_id;
        ELSE
        	SELECT unsorted_wastes INTO @previous_weight FROM user WHERE card_id = NEW.user_id;
            UPDATE user SET unsorted_wastes = @previous_weight + NEW.garbage_weight WHERE card_id = NEW.user_id;
        END IF;
    ELSE
    	UPDATE bin SET capacity = 0 WHERE id = NEW.bin_id;
    END IF;
END
$$
DELIMITER ;

--
-- Indici per le tabelle scaricate
--

--
-- Indici per le tabelle `bin`
--
ALTER TABLE `bin`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`card_id`);

--
-- Indici per le tabelle `user_interaction`
--
ALTER TABLE `user_interaction`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `bin_id` (`bin_id`);

--
-- AUTO_INCREMENT per le tabelle scaricate
--

--
-- AUTO_INCREMENT per la tabella `user_interaction`
--
ALTER TABLE `user_interaction`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=40;

--
-- Limiti per le tabelle scaricate
--

--
-- Limiti per la tabella `user_interaction`
--
ALTER TABLE `user_interaction`
  ADD CONSTRAINT `bin_id` FOREIGN KEY (`bin_id`) REFERENCES `bin` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `user_id` FOREIGN KEY (`user_id`) REFERENCES `user` (`card_id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
