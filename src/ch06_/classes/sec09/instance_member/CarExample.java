package ch06_.classes.sec09.instance_member;
//main methods signature :  public static void main(String[] args)
/* public: 모든 외부에서 접근 가능하게 합니다.
static: 인스턴스를 생성하지 않고 클래스 레벨에서 메소드에 접근할 수 있게 합니다.
void: 반환 값이 없습니다.
//String[] args: 명령줄 인수를 받을 수 있는 매개변수입니다.*/

public class CarExample {
	public static void main(String[] args) {
		Car myCar = new Car("포르쉐"); // 인스턴스화 되었다는 의미는 반드시 new 사용하여 고유한 
		// 인스턴스 값을 가짐
		Car yourCar = new Car("벤츠");

		myCar.run();
		yourCar.run();
	}
}