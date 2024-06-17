
package ch06_.classes.sec07.constructor.exam02.korean;
// 변수 6개 선언필요 
public class KoreanExample2 {
	public static void main(String[] args) {
		//Korean 객체 생성
		String k1_nation = "대한민국";
		String k1_name = "박자바";
		String k1_ssn = "011225-1234567";
	
		//Korean 객체 데이터 읽기
		System.out.println("k1.nation : " + k1_nation);
		System.out.println("k1.name : " + k1_name);
		System.out.println("k1.ssn : " + k1_ssn);
		System.out.println();

		String k2_nation = "대한민국";
		String k2_name = "김자바";
		String k2_ssn = "930525-0654321";
		//또 다른 Korean 객체 생성
		//또 다른 Korean 객체 데이터 읽기
		System.out.println("k2.nation : " + k2_nation);
		System.out.println("k2.name : " + k2_name);
		System.out.println("k2.ssn : " + k2_ssn);
	}
}
