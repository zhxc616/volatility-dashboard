import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from analysis import calculate_volatility, get_company_info, fetch_and_save_data

class TestFinancialDashboard(unittest.TestCase):

    # 1. Test the Math (Volatility Calculation)
    @patch('analysis.sqlite3')  # Mock the database connection
    @patch('analysis.pd.read_sql')  # Mock the pandas SQL reader
    def test_calculate_volatility(self, mock_read_sql, mock_sqlite):
        """
        Test if volatility is calculated correctly for known data.
        """
        # Create fake data: Stock price goes up 1% every day for 252 days
        dates = pd.date_range(start='2023-01-01', periods=252)
        prices = [100 * (1.01 ** i) for i in range(252)]

        mock_df = pd.DataFrame({'Date': dates, 'Close': prices})
        mock_read_sql.return_value = mock_df

        # Run the function
        vol = calculate_volatility('FAKE')


        self.assertTrue(vol > 0)
        self.assertIsInstance(vol, float)

    # 2. Test Data Fetching (Success Case)
    @patch('analysis.yf.Ticker')
    @patch('analysis.sqlite3')
    def test_fetch_data_success(self, mock_sqlite, mock_ticker):
        """
        Test that data is correctly processed from Yahoo.
        """
        # Mock the Yahoo Finance response
        mock_stock = MagicMock()
        mock_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Close': [100 + i for i in range(10)]
        })
        mock_stock.history.return_value = mock_df
        mock_ticker.return_value = mock_stock

        # Run function (should NOT raise an error)
        try:
            fetch_and_save_data('TSLA')
        except ValueError:
            self.fail("fetch_and_save_data raised ValueError unexpectedly!")

    # 3. Test Data Fetching (Failure Case - Empty Data)
    @patch('analysis.yf.Ticker')
    def test_fetch_data_empty(self, mock_ticker):
        """
        Test that we raise a ValueError when Yahoo returns nothing.
        """
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()  # Empty DF
        mock_ticker.return_value = mock_stock

        # We expect this specific error
        with self.assertRaises(ValueError):
            fetch_and_save_data('INVALID')

    # 4. Test Formatting (Market Cap)
    @patch('analysis.yf.Ticker')
    def test_company_info_formatting(self, mock_ticker):
        """
        Test if large numbers are formatted into B/T/M strings correctly.
        """
        mock_info = {
            'sector': 'Technology',
            'marketCap': 2500000000,  # 2.5 Billion
            'trailingPE': 25.5,
            'fiftyTwoWeekHigh': 150.00
        }
        mock_stock = MagicMock()
        mock_stock.info = mock_info
        mock_ticker.return_value = mock_stock

        info = get_company_info('AAPL')

        self.assertEqual(info['market_cap'], "$2.50B")  # Should format to billions
        self.assertEqual(info['sector'], "Technology")


if __name__ == '__main__':
    unittest.main()