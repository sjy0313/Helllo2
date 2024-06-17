package ch06_.classes.sec10.static_member.exam01.static_field_method;
/* 
 * static member(정적멤버)
 * -메소드 영역의 클래스에 고정적으로 위치하는 멤버
 * -객체를 생성할 필요 없이 클래스를 통해 바로 사용가능
 * -클래스가 사용되기 위해 로딩될 떄 미리 생성됨
 * -클래스별로 하나만 존재 
 * -런타임 실행환경에서 유일함.
 */
public class Calculator {
	static double pi = 3.14159;

	static int plus(int x, int y) {
		return x + y;
	}
	
	static int minus(int x, int y) {
		return x - y;
	}
}