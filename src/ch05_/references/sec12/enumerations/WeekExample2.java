
package ch05_.references.sec12.enumerations;

import java.util.Calendar;

public class WeekExample2 {
	public static void main(String[] args) {
		//Week 열거 타입 변수 선언
		Week today = null;
 
		//Calendar 얻기
		Calendar cal = Calendar.getInstance();
		
		//오늘의 요일 얻기(1~7)
		int week = cal.get(Calendar.DAY_OF_WEEK);

		//숫자를 열거 상수로 변환해서 변수에 대입
		today = switch(week) {
			case 1 -> Week.SUNDAY ;
			case 2 -> Week.MONDAY ; 
			case 3 -> Week.TUESDAY ; 
			case 4 -> Week.WEDNESDAY ;
			case 5 -> Week.THURSDAY ;
			case 6 -> Week.FRIDAY ; 
			case 7 -> Week.SATURDAY ;
			default -> Week.SUNDAY;
		}; // 하나의 (표현식)명령문으로 보기떄문에 }뒤에 원래는 ;오지 않지만 붙힘.
		
		//열거 타입 변수를 사용
		if(today == Week.SUNDAY) {
			System.out.println("일요일에는 축구를 합니다.");
		
		} else {
			System.out.println("열심히 자바를 공부합니다.");
		}
	}
}