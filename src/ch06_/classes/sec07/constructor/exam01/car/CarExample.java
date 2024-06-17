package ch06_.classes.sec07.constructor.exam01.car;

public class CarExample {
	public static void main(String[] args) {
	// static = 인스턴스를 생성하지 않고 클래스 레벨에서 메소드에 접근할 수 있게 함
	// void: 반환 값이 없습니다.
	// String[] args: 명령줄 인수를 받을 수 있는 매개변수
		// 클래스 변수 = new 클래스();[생성자 호출]
		Car myCar = new Car("그랜저", "검정", 250);
		// The constructor Car() is undefined : class 의 생성자 정의 되어 있으면 기본 생성자는 자동으로 만들어지지 않는다
		
		//Car myCar2 = new Car();  //기본 생성자 호출 못함
	}
}




