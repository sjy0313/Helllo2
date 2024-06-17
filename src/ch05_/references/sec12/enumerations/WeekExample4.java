
package ch05_.references.sec12.enumerations;
// weekkorean 참조
import java.util.Calendar;

public class WeekExample4 {
	public static void main(String[] args) {
		//Week 열거 타입 변수 선언
		Week today = null;
		
		//Calendar 얻기
		Calendar cal = Calendar.getInstance();
		
		int year = cal.get(Calendar.YEAR);
		int month = cal.get(Calendar.MONDAY) + 1; // 0:1월
 		int day = cal.get(Calendar.DAY_OF_MONTH);
		int week = cal.get(Calendar.DAY_OF_WEEK); // 1: 일요일 / 2: 월요일, ...
 		
		Week[] weeks = Week.values();
		WeekKorean[] weekkors = WeekKorean.values();
  		System.out.printf("오늘 : (%d)년 (%d)월 (%d)일 (%d)(%s)", year, month, day, week, weekkors[week-1]);
  		
  		for(WeekKorean wk : weekkors) {
  		System.out.printf("WeekKorean : name(%s), ordinal(%d)\n",wk.name(), wk.ordinal()); 
  		}
  		
  		
	}
}