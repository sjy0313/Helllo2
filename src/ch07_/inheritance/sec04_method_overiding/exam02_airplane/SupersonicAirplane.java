package ch07_.inheritance.sec04_method_overiding.exam02_airplane;

public class SupersonicAirplane extends Airplane {
	//상수 선언
	public static final int NORMAL = 1;
	public static final int SUPERSONIC = 2;
	//상태 필드 선언
	public int flyMode = NORMAL;

	//메소드 재정의
	// 부모메서드는 숨겨지고 자식 메소드만 사용됨
	// 따라서 일부만 변경된다 하더라도 중복된 내용을 자식 메소드도 가지고 있어야함
	// 이문제는 자식메소드내에 super.method(); 활용을 통해 부모메서드를 호출가능
	// super.method()는 
	
	@Override // 생략가능하지만 컴파일 단계에서 오버라이딩이 되었는지 
	//체크하고 문제발생시 애러발생
	public void fly() {
		if(flyMode == SUPERSONIC) {
			System.out.println("초음속 비행합니다.");
		} else {
			// 부모인 Airplane 객체의 fly() 메소드 호출
			super.fly();
			// 자신의 메소드를 호출하여 계속 반복 : stack overflow 오류
			// fly(); 
		}
	}
}