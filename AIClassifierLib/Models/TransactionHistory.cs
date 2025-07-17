namespace AIClassifierLib.Models
{
    public class TransactionHistory
    {
        public string UserId { get; set; }
        public int Hour { get; set; }
        public string DayOfWeek { get; set; }
        public string Item { get; set; }
    }
}
