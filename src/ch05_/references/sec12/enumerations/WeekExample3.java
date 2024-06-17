
package ch05_.references.sec12.enumerations;
// Week2.java 참조
import java.util.Calendar;

public class WeekExample3 {
	public static void main(String[] args) {
		// week 열거 타입 변수 선언
		//Week today = 7;
		//System.out.println("Week2: " + today);
		int now = Week2.SUNDAY;
		System.out.println("Week2: " + now);
		
		int now2 = Week2.FRIDAY; 
		System.out.println("Week2: " + now2);
	}
}